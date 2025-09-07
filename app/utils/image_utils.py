"""
Image Utility Functions for Coin Recognition

This module provides utility functions for handling images in the Coin Recognition Application.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


def cv_to_qimage(cv_image: np.ndarray) -> QImage:
    """
    Convert an OpenCV image (numpy array) to a QImage.
    
    Args:
        cv_image: OpenCV image (numpy array)
        
    Returns:
        QImage representation of the image
    """
    # Check if the image is grayscale
    if len(cv_image.shape) == 2:
        height, width = cv_image.shape
        bytes_per_line = width
        return QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    
    # Color image
    height, width, channels = cv_image.shape
    bytes_per_line = channels * width
    
    # OpenCV stores images in BGR format, but QImage expects RGB
    if channels == 3:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    return QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)


def cv_to_qpixmap(cv_image: np.ndarray) -> QPixmap:
    """
    Convert an OpenCV image (numpy array) to a QPixmap.
    
    Args:
        cv_image: OpenCV image (numpy array)
        
    Returns:
        QPixmap representation of the image
    """
    return QPixmap.fromImage(cv_to_qimage(cv_image))


def qimage_to_cv(qimage: QImage) -> np.ndarray:
    """
    Convert a QImage to an OpenCV image (numpy array).
    
    Args:
        qimage: QImage to convert
        
    Returns:
        OpenCV image (numpy array)
    """
    qimage = qimage.convertToFormat(QImage.Format_RGB888)
    width = qimage.width()
    height = qimage.height()
    
    # Get the pointer to the data
    ptr = qimage.bits()
    ptr.setsize(qimage.byteCount())
    
    # Create numpy array from data
    cv_image = np.array(ptr).reshape(height, width, 3)
    
    # Convert from RGB to BGR (OpenCV format)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    return cv_image


def resize_image(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None, 
                 keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize an image to the specified dimensions.
    
    Args:
        image: Input image
        width: Target width (if None, calculated from height and aspect ratio)
        height: Target height (if None, calculated from width and aspect ratio)
        keep_aspect_ratio: Whether to maintain the aspect ratio
        
    Returns:
        Resized image
    """
    if width is None and height is None:
        return image
    
    h, w = image.shape[:2]
    
    if keep_aspect_ratio:
        if width is None:
            # Calculate width from height
            aspect_ratio = w / h
            width = int(height * aspect_ratio)
        elif height is None:
            # Calculate height from width
            aspect_ratio = h / w
            height = int(width * aspect_ratio)
        else:
            # Both width and height are specified, but we need to maintain aspect ratio
            # We'll fit the image within the specified dimensions
            aspect_ratio = w / h
            target_aspect_ratio = width / height
            
            if aspect_ratio > target_aspect_ratio:
                # Image is wider than target, adjust height
                new_height = int(width / aspect_ratio)
                height = new_height
            else:
                # Image is taller than target, adjust width
                new_width = int(height * aspect_ratio)
                width = new_width
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def save_image(image: np.ndarray, file_path: Union[str, Path], quality: int = 95) -> bool:
    """
    Save an image to a file.
    
    Args:
        image: Image to save
        file_path: Path to save the image to
        quality: JPEG quality (0-100, higher is better)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with appropriate parameters based on file type
        if extension in ['.jpg', '.jpeg']:
            return cv2.imwrite(str(file_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif extension == '.png':
            return cv2.imwrite(str(file_path), image, [cv2.IMWRITE_PNG_COMPRESSION, min(9, quality // 10)])
        else:
            return cv2.imwrite(str(file_path), image)
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def get_image_info(image: np.ndarray) -> dict:
    """
    Get information about an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information
    """
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    
    info = {
        'width': width,
        'height': height,
        'channels': channels,
        'size': image.size,
        'dtype': str(image.dtype)
    }
    
    return info


def list_supported_image_formats() -> List[str]:
    """
    List supported image formats.
    
    Returns:
        List of supported image extensions
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def is_supported_image(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is a supported image format.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a supported image format, False otherwise
    """
    file_path = Path(file_path)
    return file_path.suffix.lower() in list_supported_image_formats()