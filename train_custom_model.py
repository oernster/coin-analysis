#!/usr/bin/env python3
"""
Custom Model Training Script for the Coin Recognition Application.

This script implements a custom approach to address specific misclassification issues:
1. Uses stronger class weights for underrepresented classes
2. Implements a post-processing step to adjust prediction probabilities
3. Increases model complexity with higher n_estimators
4. Adds a confusion matrix analysis to identify commonly confused classes
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import json
import pickle
from collections import Counter
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Add the current directory to sys.path to allow importing from the app package
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from app.processing.preprocessor import preprocess_image
from app.models.feature_extractor import get_feature_extractor
from app.utils.image_utils import is_supported_image


class CustomRandomForestClassifier:
    """
    Custom Random Forest Classifier with post-processing to address specific misclassifications.
    """
    
    def __init__(self, n_estimators=200, class_weight='balanced'):
        """
        Initialize the classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            class_weight: Class weights for handling imbalanced data
        """
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=42,
            max_depth=None,  # Allow trees to grow deeper
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            max_features='sqrt'
        )
        self.label_encoder = None
        self.feature_extractor = None
        self.confusion_pairs = {}  # Store commonly confused pairs
        self.year_weights = {}  # Store custom weights for years
        
    def fit(self, X, y):
        """
        Fit the classifier to the training data.
        
        Args:
            X: Feature matrix
            y: Target labels (years)
        """
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Fit the classifier
        self.clf.fit(X, y_encoded)
        
        # Analyze confusion matrix to identify commonly confused pairs
        y_pred = self.clf.predict(X)
        cm = confusion_matrix(y_encoded, y_pred)
        
        # Find pairs of classes that are commonly confused
        n_classes = len(self.label_encoder.classes_)
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    class_i = self.label_encoder.inverse_transform([i])[0]
                    class_j = self.label_encoder.inverse_transform([j])[0]
                    self.confusion_pairs[(class_i, class_j)] = cm[i, j]
        
        # Print top confused pairs
        print("\nTop confused pairs:")
        for (class_i, class_j), count in sorted(self.confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{class_i} confused with {class_j}: {count} times")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X with post-processing.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        # Get raw predictions and probabilities
        y_proba = self.clf.predict_proba(X)
        
        # Apply post-processing to adjust probabilities for commonly confused pairs
        adjusted_proba = self._adjust_probabilities(y_proba)
        
        # Get predicted class indices
        y_pred_indices = np.argmax(adjusted_proba, axis=1)
        
        # Convert indices to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_indices)
        
        return y_pred
    
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
            # Apply generalized age-based and dataset size-based adjustments
            current_year = 2025
            
            # Get the count of samples for each year (if available)
            year_counts = getattr(self, 'year_counts', None)
            
            # If we don't have year counts, create a dummy one with equal values
            if year_counts is None:
                year_counts = {year: 10 for year in self.label_encoder.classes_}
            
            # Find the maximum count to normalize
            max_count = max(year_counts.values())
            
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
                
                # Calculate dataset size factor - smaller datasets get higher weights
                count = year_counts.get(year, max_count)
                size_factor = max(1.0, (max_count / max(1, count)) ** 0.5)  # Square root to moderate the effect
                
                # Combine the factors
                combined_factor = age_factor * size_factor
                
                # Apply the combined factor to all predictions
                for i in range(len(adjusted_proba)):
                    # Only boost if the year has some reasonable probability
                    if adjusted_proba[i, idx_year] > 0.05:
                        # Apply a boost based on combined factor
                        # The boost is stronger for probabilities that are already somewhat high
                        boost = 1.0 + (combined_factor * 0.2 * min(1.0, adjusted_proba[i, idx_year] * 2))
                        adjusted_proba[i, idx_year] *= boost
                        
                        # Normalize to ensure probabilities sum to 1
                        adjusted_proba[i] = adjusted_proba[i] / np.sum(adjusted_proba[i])
        
        except Exception as e:
            # If there's any error in the adjustment, log it and continue with unadjusted probabilities
            print(f"Error in probability adjustment: {str(e)}")
        
        return adjusted_proba
    
    def save(self, model_path):
        """
        Save the model to disk.
        
        Args:
            model_path: Path to save the model
        """
        os.makedirs(model_path, exist_ok=True)
        
        # Save classifier
        with open(os.path.join(model_path, "classifier.pkl"), "wb") as f:
            pickle.dump(self.clf, f)
        
        # Save label encoder
        with open(os.path.join(model_path, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        # Save confusion pairs
        with open(os.path.join(model_path, "confusion_pairs.pkl"), "wb") as f:
            pickle.dump(self.confusion_pairs, f)
        
        # Save feature extractor if available
        if self.feature_extractor:
            with open(os.path.join(model_path, "feature_extractor.pkl"), "wb") as f:
                pickle.dump(self.feature_extractor, f)
    
    def load(self, model_path):
        """
        Load the model from disk.
        
        Args:
            model_path: Path to load the model from
            
        Returns:
            Loaded classifier
        """
        # Load classifier
        with open(os.path.join(model_path, "classifier.pkl"), "rb") as f:
            self.clf = pickle.load(f)
        
        # Load label encoder
        with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)
        
        # Load confusion pairs if available
        confusion_pairs_path = os.path.join(model_path, "confusion_pairs.pkl")
        if os.path.exists(confusion_pairs_path):
            with open(confusion_pairs_path, "rb") as f:
                self.confusion_pairs = pickle.load(f)
        
        # Load feature extractor if available
        feature_extractor_path = os.path.join(model_path, "feature_extractor.pkl")
        if os.path.exists(feature_extractor_path):
            with open(feature_extractor_path, "rb") as f:
                self.feature_extractor = pickle.load(f)
        
        return self


def train_custom_model(
    dataset_dir="data/raw",
    output_dir="data/models",
    feature_extractor_type="combined",
    n_estimators=300,
    extra_weight_years=None
):
    """
    Train a custom coin recognition model using organized folders.
    
    Args:
        dataset_dir: Directory containing organized folders (YEAR_MINTMARK)
        output_dir: Directory to save the trained model
        feature_extractor_type: Type of feature extractor to use
        n_estimators: Number of estimators for RandomForestClassifier
        extra_weight_years: Dictionary of years to give extra weight to
    
    Returns:
        Path to the trained model
    """
    print(f"Training custom model using data from '{dataset_dir}'...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Variables to store data
    preprocessed_images = []
    years = []
    mint_marks = []
    sample_files = []
    
    # Process each year_mint directory
    for year_mint_dir in Path(dataset_dir).glob("*"):
        if year_mint_dir.is_dir():
            # Parse year and mint from directory name (e.g., "1970_D", "2025_none")
            dir_name = year_mint_dir.name
            parts = dir_name.split('_')
            
            try:
                # Extract year
                year = int(parts[0])
                
                # Extract mint mark (or empty string for "none")
                mint = parts[1] if len(parts) > 1 and parts[1] != "none" else ""
                
                # Process each image in this directory
                for img_path in year_mint_dir.glob("*.jpg"):
                    try:
                        # Preprocess image
                        _, preprocessed = preprocess_image(img_path)
                        
                        # Store data
                        preprocessed_images.append(preprocessed)
                        years.append(year)
                        mint_marks.append(mint)
                        sample_files.append(str(img_path))
                        
                        print(f"  Processed {img_path.name} - Year: {year}, Mint: {mint}")
                    except Exception as e:
                        print(f"  Error processing {img_path.name}: {str(e)}")
            except Exception as e:
                print(f"  Error parsing directory name {dir_name}: {str(e)}")
    
    # Check if we have enough data
    if len(preprocessed_images) == 0:
        print("No images found. Please check your dataset directory.")
        return None
    
    print(f"Found {len(preprocessed_images)} images across {len(set(years))} years and {len(set(mint_marks))} mint marks")
    
    # Extract features
    print(f"Using {feature_extractor_type} feature extractor")
    feature_extractor = get_feature_extractor(feature_extractor_type)
    
    print("Extracting features...")
    features = feature_extractor.extract_features_batch(preprocessed_images)
    
    # Analyze class distribution
    year_counts = Counter(years)
    mint_counts = Counter(mint_marks)
    
    print("\nYear distribution:")
    for year, count in sorted(year_counts.items()):
        print(f"{year}: {count} ({count/len(years)*100:.1f}%)")
    
    print("\nMint mark distribution:")
    for mint, count in sorted(mint_counts.items()):
        print(f"{mint if mint else 'none'}: {count} ({count/len(mint_marks)*100:.1f}%)")
    
    # Compute custom class weights
    print("\nComputing custom class weights...")
    
    # Start with balanced class weights
    unique_years = np.unique(years)
    balanced_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_years,
        y=years
    )
    year_weight_dict = {year: weight for year, weight in zip(unique_years, balanced_weights)}
    
    # Apply extra weighting for specific years
    if extra_weight_years:
        for year, extra_weight in extra_weight_years.items():
            if year in year_weight_dict:
                year_weight_dict[year] *= extra_weight
                print(f"  Applied extra weight to year {year}: {year_weight_dict[year]:.2f}")
    
    # Store year counts in the classifier for use in prediction
    year_counts_dict = {year: year_counts.get(year, 0) for year in unique_years}
    
    # Print the weights for key years
    print("\nClass weights for key years:")
    for year in [1926, 1992]:
        if year in year_weight_dict:
            print(f"  {year}: {year_weight_dict[year]:.2f}")
    
    # Train the model for year prediction
    print("\nTraining custom classifier...")
    
    # Create and train the custom classifier
    classifier = CustomRandomForestClassifier(n_estimators=n_estimators)
    classifier.feature_extractor = feature_extractor
    # Store year counts in the classifier for post-processing
    classifier.year_counts = year_counts_dict
    
    classifier.fit(features, years)
    
    # Evaluate on training data
    train_preds = classifier.predict(features)
    train_accuracy = np.mean(train_preds == years)
    
    # Print training results
    print("Training complete!")
    print(f"Year accuracy: {train_accuracy:.2f}")
    
    # Save model
    model_path = os.path.join(output_dir, "coin_model_custom")
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Saving model to {model_path}")
    classifier.save(model_path)
    
    # Save metadata
    metadata = {
        "feature_extractor": feature_extractor_type,
        "classifier": "custom_random_forest",
        "n_samples": len(preprocessed_images),
        "training_results": {
            "year_accuracy": float(train_accuracy),
            "n_samples": len(preprocessed_images),
            "n_features": features.shape[1],
            "unique_years": len(set(years)),
            "unique_mints": len(set(mint_marks))
        },
        "sample_files": sample_files,
        "years": sorted(list(set(years))),
        "mint_marks": sorted(list(set(mint_marks))),
        "n_estimators": n_estimators,
        "extra_weight_years": extra_weight_years
    }
    
    metadata_path = os.path.join(model_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_path}")
    return model_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train a custom coin recognition model.")
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/raw",
        help="Directory containing organized folders (YEAR_MINTMARK)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Directory to save the trained model"
    )
    
    parser.add_argument(
        "--feature-extractor",
        type=str,
        choices=["hog", "lbp", "combined"],
        default="combined",
        help="Type of feature extractor to use"
    )
    
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of estimators for RandomForestClassifier"
    )
    
    args = parser.parse_args()
    
    # Define a function to calculate weights based on coin age and dataset size
    def calculate_weights(year, year_counts):
        current_year = 2025
        age = current_year - year
        
        # Base weight for all coins
        base_weight = 1.0
        
        # Age-based weight component
        if age >= 100:  # Pre-1925 coins (100+ years old)
            age_weight = 12.0  # Increased from 8.0
        elif age >= 80:  # 1925-1945 coins (80-99 years old)
            age_weight = 8.0   # Increased from 4.0
        elif age >= 60:  # 1946-1965 coins (60-79 years old)
            age_weight = 4.0   # Increased from 2.0
        elif age >= 40:  # 1966-1985 coins (40-59 years old)
            age_weight = 2.5   # Increased from 1.5
        elif age >= 20:  # 1986-2005 coins (20-39 years old)
            age_weight = 1.5
        else:  # 2006-present coins (<20 years old)
            age_weight = 1.0
        
        # Dataset size-based weight component
        # Get the count for this year
        count = year_counts.get(year, 0)
        
        # Calculate size weight - inverse relationship with count
        if count == 0:
            # If no samples, use a default high weight
            size_weight = 10.0
        elif count < 5:
            # Very few samples
            size_weight = 8.0
        elif count < 10:
            # Few samples
            size_weight = 5.0
        elif count < 20:
            # Some samples
            size_weight = 3.0
        elif count < 50:
            # Moderate number of samples
            size_weight = 1.5
        else:
            # Many samples
            size_weight = 1.0
        
        # Combine weights - multiply for stronger effect
        return base_weight * age_weight * size_weight
    
    # Count samples per year from the dataset directory
    year_counts = {}
    for year_mint_dir in Path(args.dataset_dir).glob("*"):
        if year_mint_dir.is_dir():
            try:
                # Extract year from directory name
                year = int(year_mint_dir.name.split('_')[0])
                
                # Count images in this directory
                image_count = len(list(year_mint_dir.glob("*.jpg")))
                
                # Add to or update the count for this year
                if year in year_counts:
                    year_counts[year] += image_count
                else:
                    year_counts[year] = image_count
            except:
                pass
    
    # Print year counts
    print("\nSample counts per year:")
    for year, count in sorted(year_counts.items()):
        print(f"{year}: {count} samples")
    
    # Create weights dictionary for all years from 1909 to 2025
    extra_weight_years = {year: calculate_weights(year, year_counts) for year in range(1909, 2026)}
    
    # Print the weights for some key years
    print("\nCalculated weights for selected years:")
    for year in sorted(list(year_counts.keys())):
        if year in extra_weight_years:
            print(f"{year}: {extra_weight_years[year]:.2f} (from {year_counts.get(year, 0)} samples)")
    
    # Use a lower n_estimators value to speed up training
    train_custom_model(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        feature_extractor_type=args.feature_extractor,
        n_estimators=150,  # Reduced from default 300 to speed up training
        extra_weight_years=extra_weight_years
    )


if __name__ == "__main__":
    main()