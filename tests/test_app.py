"""
Basic tests for the Coin Recognition Application.

This module contains tests to verify the basic functionality of the application.
"""

import sys
import os
import unittest
from pathlib import Path

# Add the parent directory to sys.path to allow importing from the app package
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import numpy as np
import cv2

from app.processing.preprocessor import (
    convert_to_grayscale, enhance_contrast, reduce_noise, 
    detect_edges, extract_roi, preprocess_image
)
from app.models.feature_extractor import (
    HOGFeatureExtractor, LBPFeatureExtractor, CombinedFeatureExtractor
)
from app.utils.image_utils import (
    is_supported_image, list_supported_image_formats
)


class TestPreprocessing(unittest.TestCase):
    """Test image preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some features to the image
        self.test_image[30:70, 30:70] = 255
        
        # Save the test image to a temporary file
        self.test_image_path = os.path.join(parent_dir, 'tests', 'test_image.png')
        cv2.imwrite(self.test_image_path, self.test_image)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary test image
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def test_convert_to_grayscale(self):
        """Test grayscale conversion."""
        gray = convert_to_grayscale(self.test_image)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape[0], 100)
        self.assertEqual(gray.shape[1], 100)
    
    def test_enhance_contrast(self):
        """Test contrast enhancement."""
        gray = convert_to_grayscale(self.test_image)
        enhanced = enhance_contrast(gray)
        self.assertEqual(enhanced.shape, gray.shape)
    
    def test_reduce_noise(self):
        """Test noise reduction."""
        gray = convert_to_grayscale(self.test_image)
        denoised = reduce_noise(gray)
        self.assertEqual(denoised.shape, gray.shape)
    
    def test_detect_edges(self):
        """Test edge detection."""
        gray = convert_to_grayscale(self.test_image)
        edges = detect_edges(gray)
        self.assertEqual(edges.shape, gray.shape)
    
    def test_extract_roi(self):
        """Test ROI extraction."""
        roi = extract_roi(self.test_image, 'right_half')
        self.assertEqual(roi.shape[0], 100)
        self.assertEqual(roi.shape[1], 50)
    
    def test_preprocess_image(self):
        """Test complete preprocessing pipeline."""
        original, preprocessed = preprocess_image(self.test_image_path)
        self.assertIsNotNone(original)
        self.assertIsNotNone(preprocessed)


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image
        self.test_image = np.zeros((128, 128), dtype=np.uint8)
        # Add some features to the image
        self.test_image[30:70, 30:70] = 255
    
    def test_hog_feature_extractor(self):
        """Test HOG feature extraction."""
        extractor = HOGFeatureExtractor()
        features = extractor.extract_features(self.test_image)
        self.assertIsNotNone(features)
        self.assertTrue(len(features) > 0)
    
    def test_lbp_feature_extractor(self):
        """Test LBP feature extraction."""
        extractor = LBPFeatureExtractor()
        features = extractor.extract_features(self.test_image)
        self.assertIsNotNone(features)
        self.assertTrue(len(features) > 0)
    
    def test_combined_feature_extractor(self):
        """Test combined feature extraction."""
        extractor = CombinedFeatureExtractor()
        features = extractor.extract_features(self.test_image)
        self.assertIsNotNone(features)
        self.assertTrue(len(features) > 0)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_supported_image_formats(self):
        """Test supported image formats."""
        formats = list_supported_image_formats()
        self.assertTrue(len(formats) > 0)
        self.assertIn('.jpg', formats)
        self.assertIn('.png', formats)
    
    def test_is_supported_image(self):
        """Test image format checking."""
        self.assertTrue(is_supported_image('test.jpg'))
        self.assertTrue(is_supported_image('test.png'))
        self.assertFalse(is_supported_image('test.txt'))
        self.assertFalse(is_supported_image('test.pdf'))


if __name__ == '__main__':
    unittest.main()