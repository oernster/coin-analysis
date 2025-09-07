"""
Image Preprocessing Module for Coin Recognition

This module provides functions for preprocessing coin images before recognition.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from the given path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        The loaded image as a numpy array
        
    Raises:
        ValueError: If the image cannot be loaded
    """
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    
    Args:
        image: Input image
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8), force_grayscale: bool = False) -> np.ndarray:
    """
    Enhance the contrast of an image using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        force_grayscale: Whether to force conversion to grayscale
        
    Returns:
        Contrast-enhanced image
    """
    # Convert to grayscale if needed
    working_image = convert_to_grayscale(image) if force_grayscale or len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1) else image
    
    # For color images, process each channel separately
    if len(working_image.shape) == 3 and working_image.shape[2] == 3:
        # Split the image into channels
        channels = cv2.split(working_image)
        enhanced_channels = []
        
        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        for channel in channels:
            enhanced_channels.append(clahe.apply(channel))
        
        # Merge the channels back
        enhanced_image = cv2.merge(enhanced_channels)
    else:
        # For grayscale images, apply CLAHE directly
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_image = clahe.apply(working_image)
    
    return enhanced_image


def reduce_noise(image: np.ndarray, method: str = 'gaussian', kernel_size: int = 5) -> np.ndarray:
    """
    Reduce noise in an image.
    
    Args:
        image: Input image
        method: Noise reduction method ('gaussian', 'median', or 'bilateral')
        kernel_size: Size of the kernel for filtering
        
    Returns:
        Noise-reduced image
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    else:
        raise ValueError(f"Unknown noise reduction method: {method}")


def detect_edges(image: np.ndarray, method: str = 'canny', threshold1: int = 50, threshold2: int = 150, force_grayscale: bool = False) -> np.ndarray:
    """
    Detect edges in an image.
    
    Args:
        image: Input image
        method: Edge detection method ('canny', 'sobel', or 'laplacian')
        threshold1: First threshold for Canny edge detector
        threshold2: Second threshold for Canny edge detector
        force_grayscale: Whether to force conversion to grayscale
        
    Returns:
        Edge image with the same dimensions as the input
    """
    # Store original dimensions and channels
    original_shape = image.shape
    is_color = len(original_shape) == 3 and original_shape[2] == 3
    
    # Convert to grayscale if needed or requested
    if force_grayscale or is_color:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
    
    # Apply edge detection
    if method == 'canny':
        edges = cv2.Canny(gray_image, threshold1, threshold2)
    elif method == 'sobel':
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobel_x, sobel_y)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    elif method == 'laplacian':
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        edges = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    
    # Ensure the result has the same dimensions as the input
    if edges.shape[:2] != original_shape[:2]:
        edges = cv2.resize(edges, (original_shape[1], original_shape[0]))
    
    # Convert back to color if the input was color
    if is_color and len(edges.shape) == 2:
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges


def extract_roi(image: np.ndarray, roi_type: str = 'right_half') -> np.ndarray:
    """
    Extract a region of interest (ROI) from an image.
    
    Args:
        image: Input image
        roi_type: Type of ROI to extract ('right_half', 'left_half', 'center')
        
    Returns:
        Extracted ROI
    """
    height, width = image.shape[:2]
    
    if roi_type == 'right_half':
        # Extract the right half of the image (where the year and mint mark are located)
        roi = image[:, width // 2:]
    elif roi_type == 'left_half':
        # Extract the left half of the image
        roi = image[:, :width // 2]
    elif roi_type == 'center':
        # Extract the center of the image
        center_x, center_y = width // 2, height // 2
        size = min(width, height) // 3
        roi = image[center_y - size:center_y + size, center_x - size:center_x + size]
    else:
        raise ValueError(f"Unknown ROI type: {roi_type}")
    
    return roi


def preprocess_image(image_path: Union[str, Path], roi_type: str = 'right_half',
                    apply_roi: bool = True, apply_grayscale: bool = True,
                    apply_contrast: bool = True, apply_noise_reduction: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess an image for coin recognition.
    
    Args:
        image_path: Path to the image file or a loaded image
        roi_type: Type of ROI to extract
        apply_roi: Whether to extract ROI
        apply_grayscale: Whether to convert to grayscale
        apply_contrast: Whether to enhance contrast
        apply_noise_reduction: Whether to reduce noise
        
    Returns:
        Tuple of (original image, preprocessed image)
    """
    # Load the image if it's a path
    if isinstance(image_path, (str, Path)):
        original_image = load_image(image_path)
    else:
        # If image_path is already a numpy array (loaded image)
        original_image = image_path
    
    # Start with the original image
    processed_image = original_image.copy()
    
    # Extract ROI if requested
    if apply_roi:
        processed_image = extract_roi(processed_image, roi_type)
    
    # Convert to grayscale if requested
    if apply_grayscale:
        processed_image = convert_to_grayscale(processed_image)
    
    # Enhance contrast if requested
    if apply_contrast:
        processed_image = enhance_contrast(processed_image, force_grayscale=apply_grayscale)
    
    # Reduce noise if requested
    if apply_noise_reduction:
        processed_image = reduce_noise(processed_image, method='gaussian')
    
    return original_image, processed_image


def batch_preprocess_images(image_paths: List[Union[str, Path]], roi_type: str = 'right_half',
                           apply_roi: bool = True, apply_grayscale: bool = True,
                           apply_contrast: bool = True, apply_noise_reduction: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Preprocess multiple images for coin recognition.
    
    Args:
        image_paths: List of paths to image files
        roi_type: Type of ROI to extract
        apply_roi: Whether to extract ROI
        apply_grayscale: Whether to convert to grayscale
        apply_contrast: Whether to enhance contrast
        apply_noise_reduction: Whether to reduce noise
        
    Returns:
        List of tuples of (original image, preprocessed image)
    """
    results = []
    
    for image_path in image_paths:
        try:
            result = preprocess_image(
                image_path,
                roi_type=roi_type,
                apply_roi=apply_roi,
                apply_grayscale=apply_grayscale,
                apply_contrast=apply_contrast,
                apply_noise_reduction=apply_noise_reduction
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return results