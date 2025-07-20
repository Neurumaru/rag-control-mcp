#!/usr/bin/env python3
"""CLI interface for MCP-RAG-Control Agent C."""

import argparse
import sys
from pathlib import Path

import uvicorn


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="MCP-RAG-Control Agent C")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to bind to (default: 127.0.0.1)"
    )
    api_parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind to (default: 8000)"
    )
    api_parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    api_parser.add_argument(
        "--log-level", 
        default="info", 
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)"
    )
    
    # Web interface command  
    web_parser = subparsers.add_parser("web", help="Start the Streamlit web interface")
    web_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for web interface (default: 8501)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "api":
        start_api_server(args)
    elif args.command == "web":
        start_web_interface(args)


def start_api_server(args):
    """Start the FastAPI server."""
    print(f"🚀 Starting MCP-RAG-Control Agent C API Server...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔍 Alternative docs: http://{args.host}:{args.port}/redoc")
    
    try:
        uvicorn.run(
            "mcp_rag_control.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Agent C API Server...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)


def start_web_interface(args):
    """Start the Streamlit web interface."""
    print(f"🚀 Starting MCP-RAG-Control Web Interface...")
    print(f"🌐 Interface will be available at: http://localhost:{args.port}")
    
    try:
        import subprocess
        subprocess.run([
            "streamlit", "run", 
            str(Path(__file__).parent / "web" / "app.py"),
            "--server.port", str(args.port)
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Web Interface...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()