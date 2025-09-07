#!/usr/bin/env python3
"""
Augment data for underrepresented classes in the coin dataset.

This script generates additional training samples for classes with few images
by applying various transformations to the existing images.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
import shutil

def augment_image(image, output_path, num_augmentations=10):
    """
    Apply various transformations to an image to create augmented versions.
    
    Args:
        image: Input image (numpy array)
        output_path: Path to save augmented images
        num_augmentations: Number of augmented images to generate
    """
    height, width = image.shape[:2]
    
    for i in range(num_augmentations):
        # Create a copy of the original image
        augmented = image.copy()
        
        # Apply random rotation (up to 20 degrees)
        angle = random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        augmented = cv2.warpAffine(augmented, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        
        # Apply random brightness/contrast adjustment
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.uniform(-20, 20)    # Brightness
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
        
        # Apply slight Gaussian blur
        if random.random() > 0.5:
            blur_size = random.choice([3, 5])
            augmented = cv2.GaussianBlur(augmented, (blur_size, blur_size), 0)
        
        # Apply slight noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 10, augmented.shape).astype(np.uint8)
            augmented = cv2.add(augmented, noise)
        
        # Save the augmented image
        output_file = os.path.join(output_path, f"augmented_{i+1}.jpg")
        cv2.imwrite(output_file, augmented)
        print(f"Saved augmented image: {output_file}")

def augment_underrepresented_classes(min_images_per_class=10):
    """
    Augment data for classes with fewer than min_images_per_class images.
    
    Args:
        min_images_per_class: Minimum number of images per class after augmentation
    """
    data_dir = Path("data/raw")
    
    # Process each year_mint directory
    for year_mint_dir in data_dir.glob("*"):
        if year_mint_dir.is_dir():
            # Count existing images
            images = list(year_mint_dir.glob("*.jpg"))
            num_images = len(images)
            
            if num_images < min_images_per_class and num_images > 0:
                print(f"\nAugmenting {year_mint_dir.name} ({num_images} images)")
                
                # Determine how many augmented images to generate per original image
                num_augmentations_per_image = (min_images_per_class - num_images) // num_images + 1
                
                # Process each image in this directory
                for img_path in images:
                    # Read the image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"Error reading {img_path}")
                        continue
                    
                    # Generate augmented images
                    augment_image(image, str(year_mint_dir), num_augmentations_per_image)

def main():
    """Main function."""
    # Create backup of original data
    backup_dir = Path("data/raw_backup")
    if not backup_dir.exists():
        print("Creating backup of original data...")
        shutil.copytree("data/raw", str(backup_dir))
        print(f"Backup created at {backup_dir}")
    
    # Augment underrepresented classes
    augment_underrepresented_classes(min_images_per_class=20)
    
    print("\nData augmentation complete!")
    print("You can now retrain the model with the augmented dataset.")

if __name__ == "__main__":
    main()