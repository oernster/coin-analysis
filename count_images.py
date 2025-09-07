#!/usr/bin/env python3
"""
Count the number of images per year in the dataset.
"""

import os
from pathlib import Path
from collections import Counter

def count_images_per_year():
    """Count the number of images per year in the dataset."""
    data_dir = Path("data/raw")
    year_counts = Counter()
    
    for year_mint_dir in data_dir.glob("*"):
        if year_mint_dir.is_dir():
            try:
                # Extract year from directory name (e.g., "1970_D", "2025_none")
                dir_name = year_mint_dir.name
                parts = dir_name.split('_')
                year = int(parts[0])
                
                # Count images in this directory
                image_count = len(list(year_mint_dir.glob("*.jpg")))
                year_counts[year] += image_count
                
                print(f"{dir_name}: {image_count} images")
            except Exception as e:
                print(f"Error processing {year_mint_dir.name}: {str(e)}")
    
    print("\nTotal images per year:")
    for year, count in sorted(year_counts.items()):
        print(f"{year}: {count}")

if __name__ == "__main__":
    count_images_per_year()