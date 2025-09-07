#!/usr/bin/env python3
"""
Coin Recognition Application - Main Entry Point

This module serves as the entry point for the Coin Recognition Application.
It initializes the application and shows the main window.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to allow importing from the app package
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from PySide6.QtWidgets import QApplication
from app.ui.main_window import MainWindow


def main():
    """
    Main entry point for the application.
    Initializes the QApplication and shows the main window.
    """
    app = QApplication(sys.argv)
    app.setApplicationName("Coin Recognition")
    app.setOrganizationName("CoinAnalysis")
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()