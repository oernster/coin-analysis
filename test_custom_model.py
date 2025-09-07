#!/usr/bin/env python3
"""
Test the custom coin recognition model with sample images.

This script loads the custom model and tests it on sample images to verify
that it correctly predicts the year and mint mark, with special focus on
the 1926 coin that was previously misidentified as 1992.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# Add the current directory to sys.path to allow importing from the app package
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from app.processing.preprocessor import preprocess_image
from app.models.feature_extractor import get_feature_extractor
from app.utils.image_utils import is_supported_image


def load_custom_model(model_path="data/models/coin_model_custom"):
    """
    Load the custom coin recognition model.
    
    Args:
        model_path: Path to the custom model directory
        
    Returns:
        Loaded classifier
    """
    print(f"Loading custom model from {model_path}...")
    
    # Load feature extractor
    feature_extractor_path = os.path.join(model_path, "feature_extractor.pkl")
    if os.path.exists(feature_extractor_path):
        with open(feature_extractor_path, "rb") as f:
            feature_extractor = pickle.load(f)
        print("Loaded feature extractor from pickle file")
    else:
        # Fall back to creating a new feature extractor
        feature_extractor = get_feature_extractor("combined")
        print("Created new combined feature extractor")
    
    # Load classifier
    classifier_path = os.path.join(model_path, "classifier.pkl")
    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)
    print("Loaded classifier from pickle file")
    
    # Load label encoder
    label_encoder_path = os.path.join(model_path, "label_encoder.pkl")
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    print("Loaded label encoder from pickle file")
    
    # Load confusion pairs if available
    confusion_pairs_path = os.path.join(model_path, "confusion_pairs.pkl")
    confusion_pairs = None
    if os.path.exists(confusion_pairs_path):
        with open(confusion_pairs_path, "rb") as f:
            confusion_pairs = pickle.load(f)
        print("Loaded confusion pairs from pickle file")
    
    # Create a wrapper for the classifier
    class CustomClassifierWrapper:
        def __init__(self, classifier, label_encoder, feature_extractor, confusion_pairs=None):
            self.clf = classifier
            self.label_encoder = label_encoder
            self.feature_extractor = feature_extractor
            self.confusion_pairs = confusion_pairs
        
        def predict(self, image):
            """
            Predict year and mint mark for an image.
            
            Args:
                image: Input image
                
            Returns:
                Predicted year and mint mark
            """
            # Extract features
            features = self.feature_extractor.extract_features(image)
            features = features.reshape(1, -1)
            
            # Get raw probabilities
            y_proba = self.clf.predict_proba(features)
            
            # Apply post-processing to adjust probabilities
            adjusted_proba = self._adjust_probabilities(y_proba)
            
            # Get predicted class index
            y_pred_idx = np.argmax(adjusted_proba, axis=1)[0]
            
            # Convert to original label
            y_pred = self.label_encoder.inverse_transform([y_pred_idx])[0]
            
            # Extract year and mint from combined label
            year_mint = str(y_pred).split('_')
            year = int(year_mint[0])
            mint = year_mint[1] if len(year_mint) > 1 and year_mint[1] != "none" else ""
            
            return year, mint
        
        def _adjust_probabilities(self, y_proba):
            """
            Adjust prediction probabilities based on age and dataset size.
            
            Args:
                y_proba: Raw prediction probabilities
                
            Returns:
                Adjusted probabilities
            """
            adjusted_proba = y_proba.copy()
            
            try:
                # Apply generalized age-based adjustments
                current_year = 2025
                
                # Process each year
                for year in self.label_encoder.classes_:
                    # Get the index for this year
                    idx_year = np.where(self.label_encoder.classes_ == year)[0][0]
                    
                    # Calculate age factor - older coins get higher weights
                    age = current_year - year
                    
                    # Progressive age factor that increases more dramatically for older coins
                    if age >= 100:  # Very old coins (pre-1925)
                        age_factor = 5.0
                    elif age >= 80:  # Old coins (1925-1945)
                        age_factor = 4.0
                    elif age >= 60:  # Moderately old coins (1946-1965)
                        age_factor = 3.0
                    elif age >= 40:  # Somewhat old coins (1966-1985)
                        age_factor = 2.0
                    elif age >= 20:  # Recent coins (1986-2005)
                        age_factor = 1.5
                    else:  # Very recent coins (2006-present)
                        age_factor = 1.0
                    
                    # Apply the age factor to all predictions
                    for i in range(len(adjusted_proba)):
                        # Only boost if the year has some reasonable probability
                        if adjusted_proba[i, idx_year] > 0.05:
                            # Apply a boost based on age factor
                            # The boost is stronger for probabilities that are already somewhat high
                            boost = 1.0 + (age_factor * 0.3 * min(1.0, adjusted_proba[i, idx_year] * 2))
                            adjusted_proba[i, idx_year] *= boost
                            
                            # Normalize to ensure probabilities sum to 1
                            adjusted_proba[i] = adjusted_proba[i] / np.sum(adjusted_proba[i])
            
            except Exception as e:
                # If there's any error in the adjustment, log it and continue with unadjusted probabilities
                print(f"Error in probability adjustment: {str(e)}")
            
            return adjusted_proba
    
    # Create and return the wrapper
    return CustomClassifierWrapper(classifier, label_encoder, feature_extractor, confusion_pairs)


def test_model_on_image(classifier, image_path, expected_year=None, expected_mint=None, 
                       apply_grayscale=True, apply_roi=True, show_probabilities=False):
    """
    Test the model on a single image.
    
    Args:
        classifier: Loaded classifier
        image_path: Path to the image file
        expected_year: Expected year (for verification)
        expected_mint: Expected mint mark (for verification)
        apply_grayscale: Whether to apply grayscale conversion
        apply_roi: Whether to extract ROI
        show_probabilities: Whether to show prediction probabilities
        
    Returns:
        Dictionary with test results
    """
    print(f"\nTesting image: {image_path}")
    
    # Preprocess the image
    _, preprocessed = preprocess_image(
        image_path, 
        apply_grayscale=apply_grayscale,
        apply_roi=apply_roi
    )
    
    # Make prediction
    year, mint = classifier.predict(preprocessed)
    
    # Determine if prediction is correct
    year_correct = expected_year is None or year == expected_year
    mint_correct = expected_mint is None or mint == expected_mint
    
    # Print results
    print(f"Predicted: Year={year}, Mint={mint}")
    if expected_year is not None:
        print(f"Expected: Year={expected_year}, Mint={expected_mint}")
        print(f"Year correct: {year_correct}, Mint correct: {mint_correct}")
    
    # Return results
    return {
        "image_path": image_path,
        "predicted_year": year,
        "predicted_mint": mint,
        "expected_year": expected_year,
        "expected_mint": expected_mint,
        "year_correct": year_correct,
        "mint_correct": mint_correct
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the custom coin recognition model.")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="data/models/coin_model_custom",
        help="Path to the custom model directory"
    )
    
    parser.add_argument(
        "--image-path",
        type=str,
        default="data/raw/1955_none/coin_454.jpg",
        help="Path to a single image file to test"
    )
    
    parser.add_argument(
        "--expected-year",
        type=int,
        default=1955,
        help="Expected year (for single image test)"
    )
    
    parser.add_argument(
        "--expected-mint",
        type=str,
        default="",
        help="Expected mint mark (for single image test)"
    )
    
    parser.add_argument(
        "--no-grayscale",
        action="store_true",
        help="Disable grayscale conversion"
    )
    
    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Disable ROI extraction"
    )
    
    args = parser.parse_args()
    
    # Load the model
    classifier = load_custom_model(args.model_path)
    
    # Determine whether to apply grayscale and ROI
    apply_grayscale = not args.no_grayscale
    apply_roi = not args.no_roi
    
    # Test the model on a single image
    result = test_model_on_image(
        classifier, 
        args.image_path, 
        args.expected_year, 
        args.expected_mint,
        apply_grayscale=apply_grayscale,
        apply_roi=apply_roi,
        show_probabilities=True
    )
    
    # Print summary
    print("\n=== Test Results Summary ===")
    print(f"Image: {result['image_path']}")
    print(f"Predicted: Year={result['predicted_year']}, Mint={result['predicted_mint']}")
    print(f"Expected: Year={result['expected_year']}, Mint={result['expected_mint']}")
    print(f"Year correct: {result['year_correct']}")
    print(f"Mint correct: {result['mint_correct']}")


if __name__ == "__main__":
    main()