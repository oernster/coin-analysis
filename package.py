#!/usr/bin/env python3
"""
Packaging script for the Coin Recognition Application.

This script creates a standalone executable using PyInstaller.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def package_application():
    """Package the application into a standalone executable."""
    print("Packaging Coin Recognition Application...")
    
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Clean up previous builds
    dist_dir = Path("dist")
    build_dir = Path("build")
    spec_file = Path("coin_recognition.spec")
    
    if dist_dir.exists():
        print("Removing previous dist directory...")
        shutil.rmtree(dist_dir)
    
    if build_dir.exists():
        print("Removing previous build directory...")
        shutil.rmtree(build_dir)
    
    if spec_file.exists():
        print("Removing previous spec file...")
        spec_file.unlink()
    
    # Create resource directories in the package
    print("Creating resource directories...")
    os.makedirs("dist/data/raw", exist_ok=True)
    os.makedirs("dist/data/processed", exist_ok=True)
    os.makedirs("dist/data/models", exist_ok=True)
    
    # Copy sample images
    samples_dir = Path("samples")
    if samples_dir.exists():
        print("Copying sample images...")
        os.makedirs("dist/samples", exist_ok=True)
        for sample_file in samples_dir.glob("*"):
            if sample_file.is_file():
                shutil.copy(sample_file, f"dist/samples/{sample_file.name}")
    
    # Run PyInstaller
    print("Running PyInstaller...")
    pyinstaller_args = [
        "pyinstaller",
        "--name=CoinRecognition",
        "--windowed",
        "--onedir",
        "--add-data=README.md;.",
        "--add-data=LICENSE;.",
        "--add-data=requirements.txt;.",
        "run.py"
    ]
    
    subprocess.check_call(pyinstaller_args)
    
    print("Packaging complete!")
    print(f"Executable can be found in: {os.path.abspath('dist/CoinRecognition')}")


if __name__ == "__main__":
    package_application()