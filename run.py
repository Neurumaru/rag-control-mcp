#!/usr/bin/env python3
"""
Legacy run script - use the CLI interface instead.
Run: mcp-rag-control api --help
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mcp_rag_control.cli import main

if __name__ == "__main__":
    # For backward compatibility, default to API server
    if len(sys.argv) == 1:
        sys.argv.append("api")
    main()