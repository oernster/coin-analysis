"""
Data Utility Functions for Coin Recognition

This module provides utility functions for handling data in the Coin Recognition Application.
"""

import os
import json
import csv
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
import random
import shutil


def create_dataset_directory_structure(base_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Create a directory structure for organizing dataset files.
    
    Args:
        base_dir: Base directory for the dataset
        
    Returns:
        Dictionary mapping directory names to Path objects
    """
    base_dir = Path(base_dir)
    
    # Define directory structure
    dirs = {
        'raw': base_dir / 'raw',
        'processed': base_dir / 'processed',
        'train': base_dir / 'processed' / 'train',
        'val': base_dir / 'processed' / 'val',
        'test': base_dir / 'processed' / 'test',
        'models': base_dir / 'models',
        'results': base_dir / 'results'
    }
    
    # Create directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def list_image_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    List all image files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of paths to image files
    """
    directory = Path(directory)
    
    # Get supported image extensions from image_utils
    from app.utils.image_utils import list_supported_image_formats
    extensions = list_supported_image_formats()
    
    image_files = []
    
    if recursive:
        for ext in extensions:
            image_files.extend(directory.glob(f'**/*{ext}'))
    else:
        for ext in extensions:
            image_files.extend(directory.glob(f'*{ext}'))
    
    return sorted(image_files)


def split_dataset(image_files: List[Path], train_ratio: float = 0.7, 
                 val_ratio: float = 0.15, test_ratio: float = 0.15, 
                 random_seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split a list of image files into training, validation, and test sets.
    
    Args:
        image_files: List of paths to image files
        train_ratio: Ratio of images to use for training
        val_ratio: Ratio of images to use for validation
        test_ratio: Ratio of images to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Shuffle files
    random.seed(random_seed)
    shuffled_files = image_files.copy()
    random.shuffle(shuffled_files)
    
    # Calculate split indices
    n_files = len(shuffled_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    # Split files
    train_files = shuffled_files[:n_train]
    val_files = shuffled_files[n_train:n_train + n_val]
    test_files = shuffled_files[n_train + n_val:]
    
    return train_files, val_files, test_files


def organize_dataset(image_files: List[Path], output_dirs: Dict[str, Path], 
                    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                    copy_files: bool = True) -> Dict[str, List[Path]]:
    """
    Organize a dataset by splitting it and copying/moving files to appropriate directories.
    
    Args:
        image_files: List of paths to image files
        output_dirs: Dictionary mapping split names to output directories
        split_ratios: Tuple of (train_ratio, val_ratio, test_ratio)
        copy_files: Whether to copy files (True) or move them (False)
        
    Returns:
        Dictionary mapping split names to lists of file paths
    """
    # Split dataset
    train_files, val_files, test_files = split_dataset(
        image_files, 
        train_ratio=split_ratios[0], 
        val_ratio=split_ratios[1], 
        test_ratio=split_ratios[2]
    )
    
    # Organize files
    result = {
        'train': [],
        'val': [],
        'test': []
    }
    
    # Function to copy or move files
    def transfer_file(src, dst_dir):
        dst = dst_dir / src.name
        if copy_files:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
        return dst
    
    # Process each split
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for file in files:
            dst = transfer_file(file, output_dirs[split_name])
            result[split_name].append(dst)
    
    return result


def save_dataset_metadata(metadata: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save dataset metadata to a JSON file.
    
    Args:
        metadata: Dictionary containing metadata
        output_path: Path to save the metadata to
    """
    output_path = Path(output_path)
    
    # Convert Path objects to strings
    def convert_paths(obj):
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    metadata = convert_paths(metadata)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_dataset_metadata(metadata_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dataset metadata from a JSON file.
    
    Args:
        metadata_path: Path to the metadata file
        
    Returns:
        Dictionary containing metadata
    """
    metadata_path = Path(metadata_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def save_results(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save results to a JSON file.
    
    Args:
        results: Dictionary containing results
        output_path: Path to save the results to
    """
    save_dataset_metadata(results, output_path)


def load_results(results_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load results from a JSON file.
    
    Args:
        results_path: Path to the results file
        
    Returns:
        Dictionary containing results
    """
    return load_dataset_metadata(results_path)


def save_model(model: Any, model_path: Union[str, Path]) -> None:
    """
    Save a model to a file.
    
    Args:
        model: Model to save
        model_path: Path to save the model to
    """
    model_path = Path(model_path)
    
    # Create directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path: Union[str, Path]) -> Any:
    """
    Load a model from a file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    model_path = Path(model_path)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def create_csv_file(file_path: Union[str, Path], headers: List[str]) -> None:
    """
    Create a CSV file with headers.
    
    Args:
        file_path: Path to the CSV file
        headers: List of column headers
    """
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create CSV file
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def append_to_csv(file_path: Union[str, Path], row: List[Any]) -> None:
    """
    Append a row to a CSV file.
    
    Args:
        file_path: Path to the CSV file
        row: List of values to append
    """
    file_path = Path(file_path)
    
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def read_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame
    """
    file_path = Path(file_path)
    
    return pd.read_csv(file_path)