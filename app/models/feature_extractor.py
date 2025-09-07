"""
Feature Extraction Module for Coin Recognition

This module provides functions for extracting features from preprocessed coin images.
It includes both traditional feature extraction methods and a transfer learning approach.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
import os

# For traditional feature extraction
from skimage.feature import hog, local_binary_pattern
from skimage.measure import regionprops

# For transfer learning (commented out for now to avoid dependencies)
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.models import Model


class FeatureExtractor:
    """Base class for feature extractors."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from an image.
        
        Args:
            image: Input image
            
        Returns:
            Feature vector
        """
        raise NotImplementedError("Subclasses must implement extract_features")
    
    def extract_features_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Array of feature vectors
        """
        features = []
        for image in images:
            features.append(self.extract_features(image))
        return np.array(features)


class HOGFeatureExtractor(FeatureExtractor):
    """Feature extractor using Histogram of Oriented Gradients (HOG)."""
    
    def __init__(self, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (8, 8),
                 cells_per_block: Tuple[int, int] = (2, 2), block_norm: str = 'L2-Hys'):
        """
        Initialize the HOG feature extractor.
        
        Args:
            orientations: Number of orientation bins
            pixels_per_cell: Size (in pixels) of a cell
            cells_per_block: Number of cells in each block
            block_norm: Block normalization method
        """
        super().__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from an image.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            HOG feature vector
        """
        # Ensure the image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image to a standard size
        image = cv2.resize(image, (128, 128))
        
        # Extract HOG features
        features = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=True
        )
        
        return features


class LBPFeatureExtractor(FeatureExtractor):
    """Feature extractor using Local Binary Patterns (LBP)."""
    
    def __init__(self, n_points: int = 24, radius: int = 3, method: str = 'uniform'):
        """
        Initialize the LBP feature extractor.
        
        Args:
            n_points: Number of circularly symmetric neighbor set points
            radius: Radius of circle
            method: Method to determine the pattern
        """
        super().__init__()
        self.n_points = n_points
        self.radius = radius
        self.method = method
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP features from an image.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            LBP feature vector
        """
        # Ensure the image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image to a standard size
        image = cv2.resize(image, (128, 128))
        
        # Extract LBP features
        lbp = local_binary_pattern(
            image,
            P=self.n_points,
            R=self.radius,
            method=self.method
        )
        
        # Compute histogram of LBP
        n_bins = self.n_points + 2 if self.method == 'uniform' else 2**self.n_points
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist


class CombinedFeatureExtractor(FeatureExtractor):
    """Feature extractor combining multiple feature extraction methods."""
    
    def __init__(self):
        """Initialize the combined feature extractor."""
        super().__init__()
        self.hog_extractor = HOGFeatureExtractor()
        self.lbp_extractor = LBPFeatureExtractor()
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract combined features from an image.
        
        Args:
            image: Input image
            
        Returns:
            Combined feature vector
        """
        # Extract features using different methods
        hog_features = self.hog_extractor.extract_features(image)
        lbp_features = self.lbp_extractor.extract_features(image)
        
        # Combine features
        combined_features = np.concatenate([hog_features, lbp_features])
        
        return combined_features


# Commented out to avoid TensorFlow dependency for now
# class TransferLearningFeatureExtractor(FeatureExtractor):
#     """Feature extractor using transfer learning with a pre-trained CNN."""
#     
#     def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)):
#         """
#         Initialize the transfer learning feature extractor.
#         
#         Args:
#             input_shape: Input shape for the model
#         """
#         super().__init__()
#         self.input_shape = input_shape
#         
#         # Load pre-trained model
#         base_model = MobileNetV2(
#             input_shape=input_shape,
#             include_top=False,
#             weights='imagenet'
#         )
#         
#         # Create feature extraction model
#         self.model = Model(
#             inputs=base_model.input,
#             outputs=base_model.get_layer('global_average_pooling2d').output
#         )
#     
#     def extract_features(self, image: np.ndarray) -> np.ndarray:
#         """
#         Extract features from an image using transfer learning.
#         
#         Args:
#             image: Input image (RGB)
#             
#         Returns:
#             Feature vector
#         """
#         # Ensure the image is RGB
#         if len(image.shape) == 2:
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#         elif image.shape[2] == 1:
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#         elif image.shape[2] == 3:
#             # Ensure BGR to RGB conversion if needed
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         
#         # Resize image to the input shape
#         image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
#         
#         # Preprocess image
#         image = preprocess_input(image)
#         
#         # Expand dimensions to create a batch of size 1
#         image = np.expand_dims(image, axis=0)
#         
#         # Extract features
#         features = self.model.predict(image)
#         
#         return features[0]


def get_feature_extractor(method: str = 'combined') -> FeatureExtractor:
    """
    Get a feature extractor based on the specified method.
    
    Args:
        method: Feature extraction method ('hog', 'lbp', 'combined', or 'transfer_learning')
        
    Returns:
        Feature extractor instance
    """
    if method == 'hog':
        return HOGFeatureExtractor()
    elif method == 'lbp':
        return LBPFeatureExtractor()
    elif method == 'combined':
        return CombinedFeatureExtractor()
    # elif method == 'transfer_learning':
    #     return TransferLearningFeatureExtractor()
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")