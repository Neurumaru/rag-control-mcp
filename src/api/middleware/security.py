"""Security middleware for API protection."""

import time
from typing import Dict, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import hashlib

from ...utils.logger import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent abuse."""
    
    def __init__(
        self, 
        app,
        calls_per_minute: int = 60,
        burst_limit: int = 10
    ):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.burst_limit = burst_limit
        self.client_history: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Clean old entries
        self._cleanup_old_entries(current_time)
        
        # Check rate limit
        if self._is_rate_limited(client_ip, current_time):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.calls_per_minute} requests per minute allowed",
                    "retry_after": 60
                }
            )
        
        # Record this request
        self._record_request(client_ip, current_time)
        
        response = await call_next(request)
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxies."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _cleanup_old_entries(self, current_time: float) -> None:
        """Remove entries older than 1 minute."""
        cutoff_time = current_time - 60
        
        for client_ip in list(self.client_history.keys()):
            self.client_history[client_ip] = [
                req_time for req_time in self.client_history[client_ip]
                if req_time > cutoff_time
            ]
            
            # Remove empty entries
            if not self.client_history[client_ip]:
                del self.client_history[client_ip]
    
    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client is rate limited."""
        if client_ip not in self.client_history:
            return False
        
        requests = self.client_history[client_ip]
        
        # Check burst limit (last 10 seconds)
        recent_requests = [
            req_time for req_time in requests
            if req_time > current_time - 10
        ]
        
        if len(recent_requests) >= self.burst_limit:
            return True
        
        # Check per-minute limit
        return len(requests) >= self.calls_per_minute
    
    def _record_request(self, client_ip: str, request_time: float) -> None:
        """Record a request for the client."""
        if client_ip not in self.client_history:
            self.client_history[client_ip] = []
        
        self.client_history[client_ip].append(request_time)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self'"
        )
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests for monitoring and debugging."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {client_ip} "
            f"User-Agent: {request.headers.get('user-agent', 'Unknown')}"
        )
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"Status: {response.status_code} "
            f"Time: {process_time:.3f}s"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxies."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Limit request body size to prevent abuse."""
    
    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        # Check Content-Length header
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            logger.warning(f"Request too large: {content_length} bytes from {request.client.host}")
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request too large",
                    "message": f"Maximum request size is {self.max_size} bytes",
                    "max_size": self.max_size
                }
            )
        
        response = await call_next(request)
        return response