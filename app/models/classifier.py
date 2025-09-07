"""
Classifier Module for Coin Recognition

This module provides classes for training and using classifiers for coin recognition.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
import os
import time

# For traditional ML classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Local imports
from app.models.feature_extractor import FeatureExtractor, get_feature_extractor


class CoinClassifier:
    """Base class for coin classifiers."""
    
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None):
        """
        Initialize the coin classifier.
        
        Args:
            feature_extractor: Feature extractor to use
        """
        self.feature_extractor = feature_extractor or get_feature_extractor('combined')
        self.year_classifier = None
        self.mint_classifier = None
        self.year_scaler = StandardScaler()
        self.mint_scaler = StandardScaler()
        self.trained = False
    
    def train(self, images: List[np.ndarray], years: List[int], mints: List[str]) -> Dict[str, Any]:
        """
        Train the classifier on the given data.
        
        Args:
            images: List of preprocessed coin images
            years: List of year labels
            mints: List of mint mark labels
            
        Returns:
            Dictionary with training results
        """
        raise NotImplementedError("Subclasses must implement train")
    
    def predict(self, image: np.ndarray) -> Tuple[int, str]:
        """
        Predict the year and mint mark for a coin image.
        
        Args:
            image: Preprocessed coin image
            
        Returns:
            Tuple of (predicted_year, predicted_mint)
        """
        if not self.trained:
            raise RuntimeError("Classifier has not been trained yet")
        
        # Extract features
        features = self.feature_extractor.extract_features(image)
        
        # Reshape features for sklearn
        features = features.reshape(1, -1)
        
        # Scale features
        year_features = self.year_scaler.transform(features)
        mint_features = self.mint_scaler.transform(features)
        
        # Predict
        predicted_year = self.year_classifier.predict(year_features)[0]
        predicted_mint = self.mint_classifier.predict(mint_features)[0]
        
        return predicted_year, predicted_mint
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[int, str]]:
        """
        Predict the year and mint mark for a batch of coin images.
        
        Args:
            images: List of preprocessed coin images
            
        Returns:
            List of tuples of (predicted_year, predicted_mint)
        """
        results = []
        for image in images:
            results.append(self.predict(image))
        return results
    
    def evaluate(self, images: List[np.ndarray], years: List[int], mints: List[str]) -> Dict[str, Any]:
        """
        Evaluate the classifier on the given data.
        
        Args:
            images: List of preprocessed coin images
            years: List of year labels
            mints: List of mint mark labels
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.trained:
            raise RuntimeError("Classifier has not been trained yet")
        
        # Extract features
        features = self.feature_extractor.extract_features_batch(images)
        
        # Scale features
        year_features = self.year_scaler.transform(features)
        mint_features = self.mint_scaler.transform(features)
        
        # Predict
        predicted_years = self.year_classifier.predict(year_features)
        predicted_mints = self.mint_classifier.predict(mint_features)
        
        # Calculate metrics
        year_accuracy = accuracy_score(years, predicted_years)
        mint_accuracy = accuracy_score(mints, predicted_mints)
        
        year_report = classification_report(years, predicted_years, output_dict=True)
        mint_report = classification_report(mints, predicted_mints, output_dict=True)
        
        year_cm = confusion_matrix(years, predicted_years)
        mint_cm = confusion_matrix(mints, predicted_mints)
        
        # Combine results
        results = {
            'year_accuracy': year_accuracy,
            'mint_accuracy': mint_accuracy,
            'year_report': year_report,
            'mint_report': mint_report,
            'year_confusion_matrix': year_cm.tolist(),
            'mint_confusion_matrix': mint_cm.tolist()
        }
        
        return results
    
    def save(self, model_dir: Union[str, Path]) -> None:
        """
        Save the trained classifier to a directory.
        
        Args:
            model_dir: Directory to save the model to
        """
        if not self.trained:
            raise RuntimeError("Classifier has not been trained yet")
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classifiers
        with open(model_dir / 'year_classifier.pkl', 'wb') as f:
            pickle.dump(self.year_classifier, f)
        
        with open(model_dir / 'mint_classifier.pkl', 'wb') as f:
            pickle.dump(self.mint_classifier, f)
        
        # Save scalers
        with open(model_dir / 'year_scaler.pkl', 'wb') as f:
            pickle.dump(self.year_scaler, f)
        
        with open(model_dir / 'mint_scaler.pkl', 'wb') as f:
            pickle.dump(self.mint_scaler, f)
        
        # Save metadata
        metadata = {
            'classifier_type': self.__class__.__name__,
            'feature_extractor_type': self.feature_extractor.__class__.__name__,
            'trained': self.trained,
            'saved_at': time.time()
        }
        
        with open(model_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> 'CoinClassifier':
        """
        Load a trained classifier from a directory.
        
        Args:
            model_dir: Directory to load the model from
            
        Returns:
            Loaded classifier
        """
        model_dir = Path(model_dir)
        
        # Load metadata
        with open(model_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Create classifier
        feature_extractor = get_feature_extractor('combined')
        classifier = cls(feature_extractor)
        
        # Load classifiers
        with open(model_dir / 'year_classifier.pkl', 'rb') as f:
            classifier.year_classifier = pickle.load(f)
        
        with open(model_dir / 'mint_classifier.pkl', 'rb') as f:
            classifier.mint_classifier = pickle.load(f)
        
        # Load scalers
        with open(model_dir / 'year_scaler.pkl', 'rb') as f:
            classifier.year_scaler = pickle.load(f)
        
        with open(model_dir / 'mint_scaler.pkl', 'rb') as f:
            classifier.mint_scaler = pickle.load(f)
        
        classifier.trained = metadata['trained']
        
        return classifier


class RandomForestCoinClassifier(CoinClassifier):
    """Coin classifier using Random Forest."""
    
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None,
                n_estimators: int = 100, max_depth: Optional[int] = None):
        """
        Initialize the Random Forest coin classifier.
        
        Args:
            feature_extractor: Feature extractor to use
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
        """
        super().__init__(feature_extractor)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def train(self, images: List[np.ndarray], years: List[int], mints: List[str]) -> Dict[str, Any]:
        """
        Train the classifier on the given data.
        
        Args:
            images: List of preprocessed coin images
            years: List of year labels
            mints: List of mint mark labels
            
        Returns:
            Dictionary with training results
        """
        # Extract features
        features = self.feature_extractor.extract_features_batch(images)
        
        # Scale features
        year_features = self.year_scaler.fit_transform(features)
        mint_features = self.mint_scaler.fit_transform(features)
        
        # Create and train year classifier
        self.year_classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        self.year_classifier.fit(year_features, years)
        
        # Create and train mint classifier
        self.mint_classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        self.mint_classifier.fit(mint_features, mints)
        
        # Mark as trained
        self.trained = True
        
        # Calculate training metrics
        year_accuracy = self.year_classifier.score(year_features, years)
        mint_accuracy = self.mint_classifier.score(mint_features, mints)
        
        # Return results
        results = {
            'year_accuracy': year_accuracy,
            'mint_accuracy': mint_accuracy,
            'n_samples': len(images),
            'n_features': features.shape[1],
            'unique_years': len(set(years)),
            'unique_mints': len(set(mints))
        }
        
        return results


class SVMCoinClassifier(CoinClassifier):
    """Coin classifier using Support Vector Machine."""
    
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None,
                C: float = 1.0, kernel: str = 'rbf'):
        """
        Initialize the SVM coin classifier.
        
        Args:
            feature_extractor: Feature extractor to use
            C: Regularization parameter
            kernel: Kernel type
        """
        super().__init__(feature_extractor)
        self.C = C
        self.kernel = kernel
    
    def train(self, images: List[np.ndarray], years: List[int], mints: List[str]) -> Dict[str, Any]:
        """
        Train the classifier on the given data.
        
        Args:
            images: List of preprocessed coin images
            years: List of year labels
            mints: List of mint mark labels
            
        Returns:
            Dictionary with training results
        """
        # Extract features
        features = self.feature_extractor.extract_features_batch(images)
        
        # Scale features
        year_features = self.year_scaler.fit_transform(features)
        mint_features = self.mint_scaler.fit_transform(features)
        
        # Create and train year classifier
        self.year_classifier = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=True,
            random_state=42
        )
        self.year_classifier.fit(year_features, years)
        
        # Create and train mint classifier
        self.mint_classifier = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=True,
            random_state=42
        )
        self.mint_classifier.fit(mint_features, mints)
        
        # Mark as trained
        self.trained = True
        
        # Calculate training metrics
        year_accuracy = self.year_classifier.score(year_features, years)
        mint_accuracy = self.mint_classifier.score(mint_features, mints)
        
        # Return results
        results = {
            'year_accuracy': year_accuracy,
            'mint_accuracy': mint_accuracy,
            'n_samples': len(images),
            'n_features': features.shape[1],
            'unique_years': len(set(years)),
            'unique_mints': len(set(mints))
        }
        
        return results


def get_classifier(classifier_type: str = 'random_forest', 
                  feature_extractor: Optional[FeatureExtractor] = None) -> CoinClassifier:
    """
    Get a classifier based on the specified type.
    
    Args:
        classifier_type: Type of classifier ('random_forest' or 'svm')
        feature_extractor: Feature extractor to use
        
    Returns:
        Classifier instance
    """
    if classifier_type == 'random_forest':
        return RandomForestCoinClassifier(feature_extractor)
    elif classifier_type == 'svm':
        return SVMCoinClassifier(feature_extractor)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")