"""Centralized logging configuration for MCP-RAG-Control system."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class LoggerConfig:
    """Configuration for the logging system."""
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_rotation: str = "10 MB",
        log_retention: str = "30 days",
        structured_logging: bool = True,
        enable_console: bool = True
    ):
        """Initialize logger configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file, if None uses default location
            log_rotation: When to rotate log files
            log_retention: How long to keep old log files
            structured_logging: Whether to use structured JSON logging
            enable_console: Whether to log to console
        """
        self.log_level = log_level
        self.log_file = log_file or self._get_default_log_file()
        self.log_rotation = log_rotation
        self.log_retention = log_retention
        self.structured_logging = structured_logging
        self.enable_console = enable_console
    
    def _get_default_log_file(self) -> str:
        """Get default log file path."""
        log_dir = Path.home() / ".mcp_rag_control" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir / "mcp_rag_control.log")


def setup_logging(config: Optional[LoggerConfig] = None) -> None:
    """Setup centralized logging configuration.
    
    Args:
        config: Logger configuration, uses defaults if None
    """
    if config is None:
        config = LoggerConfig()
    
    # Remove default handler
    logger.remove()
    
    # Console logging format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # File logging format (structured if enabled)
    if config.structured_logging:
        file_format = (
            "{{"
            '"timestamp": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"module": "{name}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}", '
            '"extra": {extra}'
            "}}"
        )
    else:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        )
    
    # Add console handler
    if config.enable_console:
        logger.add(
            sys.stderr,
            format=console_format,
            level=config.log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    # Add file handler
    logger.add(
        config.log_file,
        format=file_format,
        level=config.log_level,
        rotation=config.log_rotation,
        retention=config.log_retention,
        compression="gz",
        backtrace=True,
        diagnose=True,
        serialize=config.structured_logging
    )
    
    logger.info("Logging system initialized", extra={
        "log_level": config.log_level,
        "log_file": config.log_file,
        "structured_logging": config.structured_logging
    })


def get_logger(name: str) -> Any:
    """Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(module=name)


def log_function_call(func_name: str, **kwargs) -> None:
    """Log function call with parameters.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger.debug(
        f"Calling function: {func_name}",
        extra={"function_call": func_name, "parameters": kwargs}
    )


def log_error_with_context(
    error: Exception,
    context: Dict[str, Any],
    module_name: str = "unknown"
) -> None:
    """Log error with structured context information.
    
    Args:
        error: Exception that occurred
        context: Additional context information
        module_name: Name of the module where error occurred
    """
    logger.error(
        f"Error in {module_name}: {str(error)}",
        extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "module": module_name
        }
    )


def log_performance_metric(
    operation: str,
    duration_ms: float,
    success: bool = True,
    **metadata
) -> None:
    """Log performance metrics for operations.
    
    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        success: Whether operation succeeded
        **metadata: Additional metadata about the operation
    """
    logger.info(
        f"Performance metric: {operation}",
        extra={
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            "metadata": metadata
        }
    )


# Initialize logging on module import with environment-based configuration
def _init_logging_from_env():
    """Initialize logging from environment variables."""
    config = LoggerConfig(
        log_level=os.getenv("MCP_LOG_LEVEL", "INFO"),
        log_file=os.getenv("MCP_LOG_FILE"),
        log_rotation=os.getenv("MCP_LOG_ROTATION", "10 MB"),
        log_retention=os.getenv("MCP_LOG_RETENTION", "30 days"),
        structured_logging=os.getenv("MCP_STRUCTURED_LOGGING", "true").lower() == "true",
        enable_console=os.getenv("MCP_CONSOLE_LOGGING", "true").lower() == "true"
    )
    setup_logging(config)


# Auto-initialize logging
_init_logging_from_env()