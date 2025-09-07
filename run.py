#!/usr/bin/env python3
"""
Run script for the Coin Recognition Application.

This script serves as the entry point for the Coin Recognition Application.
"""

import sys
from pathlib import Path

# Add the current directory to sys.path to allow importing from the app package
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from PySide6.QtWidgets import QApplication
from app.main import main

if __name__ == "__main__":
    main()